from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mixing_copilot.io.files import atomic_json_dump, sha256_file

SCHEMA_VERSION = "1.1"
PREFERRED_EVIDENCE_STAGES = ("late_fill", "post_fill", "mid_fill", "early_fill", "pre_fill")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    value = _json_value(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return int(numeric) if integer else numeric


def _rgb(value: Any) -> list[int] | None:
    value = _json_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return [int(round(float(channel))) for channel in value[:3]]


def _relative_source_path(output_root: Path, value: Any) -> Path | None:
    value = _json_value(value)
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else output_root / path


def _copy_browser_asset(
    source: Path | None,
    image_root: Path,
    *,
    prefix: str,
    vial_id: int,
    stage: str,
) -> str | None:
    if source is None or not source.is_file():
        return None
    suffix = source.suffix.lower() or ".jpg"
    filename = f"{prefix}_vial{vial_id:04d}_{stage}{suffix}"
    destination = image_root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"/generated/images/{filename}"


def _load_predictions(predictions_root: Path) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
    if not predictions_root.is_dir():
        return predictions
    for path in sorted(predictions_root.glob("vial_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        vial_id = _number(payload.get("vial_id"), integer=True)
        if vial_id is not None:
            predictions[int(vial_id)] = payload
    return predictions


def _unavailable_ml_prediction() -> dict[str, Any]:
    return {
        "status": "unavailable",
        "model_name": "Causal TCN",
        "model_type": "causal_tcn",
        "purpose": "Predict a future rejection from early-fill visual observations.",
        "training_domain": "synthetic_isaac_sim",
        "target": None,
        "horizon_seconds": None,
        "input_shape": [1, 19, 32],
        "feature_count": 19,
        "history_frames": 32,
        "validated_on_real_csi": False,
        "score_semantics": "Model score, not a calibrated physical contamination probability.",
        "message": "Trained model available. No compatible inference sequence loaded.",
        "examples": [],
    }


def _copy_ml_evidence(
    source: Path | None, image_root: Path, example_id: str
) -> str | None:
    if source is None:
        return None
    source_path = str(source.resolve())
    if os.name == "nt" and not source_path.startswith("\\\\?\\"):
        source_path = f"\\\\?\\{source_path}"
    if not os.path.isfile(source_path):
        return None
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", example_id).strip("-") or "example"
    suffix = source.suffix.lower() or ".png"
    filename = f"ml_{safe_id}{suffix}"
    destination = image_root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return f"/generated/images/{filename}"


def _load_ml_prediction(
    path: Path, output_root: Path, image_root: Path
) -> dict[str, Any]:
    if not path.is_file():
        return _unavailable_ml_prediction()
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    model = artifact.get("model")
    examples = artifact.get("examples")
    if not isinstance(model, dict) or not isinstance(examples, list):
        raise ValueError("Causal TCN artifact must contain model and examples")
    exported_examples = []
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Causal TCN examples must be objects")
        example_id = str(example.get("example_id") or "example")
        evidence_path = _relative_source_path(
            output_root, example.get("evidence_image_path")
        )
        exported_examples.append(
            {
                "example_id": example_id,
                "source_clip": _json_value(example.get("source_clip")),
                "source_split": _json_value(example.get("source_split")),
                "camera_id": _json_value(example.get("camera_id")),
                "observation_frame": _number(
                    example.get("observation_frame"), integer=True
                ),
                "observation_timestamp_s": _number(
                    example.get("observation_timestamp_s")
                ),
                "observation_stage": _json_value(example.get("observation_stage")),
                "window_start_frame": _number(
                    example.get("window_start_frame"), integer=True
                ),
                "window_end_frame": _number(
                    example.get("window_end_frame"), integer=True
                ),
                "raw_model_logit": _number(example.get("raw_model_logit")),
                "score": _number(example.get("score")),
                "threshold": _number(example.get("threshold")),
                "decision": _json_value(example.get("decision")),
                "target": _json_value(example.get("target")),
                "actual_synthetic_label": _json_value(
                    example.get("actual_synthetic_label")
                ),
                "actual_target_label": _json_value(example.get("actual_target_label")),
                "contamination_ml": _number(example.get("contamination_ml")),
                "evidence_url": _copy_ml_evidence(
                    evidence_path, image_root, example_id
                ),
            }
        )
    if not exported_examples:
        return _unavailable_ml_prediction()
    return {
        "status": "available",
        "model_name": _json_value(model.get("model_name")) or "Causal TCN",
        "model_type": _json_value(model.get("model_type")) or "causal_tcn",
        "model_version": _json_value(model.get("model_version")),
        "purpose": _json_value(model.get("purpose")),
        "training_domain": _json_value(model.get("training_domain")),
        "target": _json_value(model.get("target")),
        "horizon_seconds": _number(model.get("horizon_seconds")),
        "input_shape": [
            _number(value, integer=True) for value in model.get("input_shape") or []
        ],
        "feature_count": _number(model.get("feature_count"), integer=True),
        "history_frames": _number(model.get("history_frames"), integer=True),
        "camera_id": _json_value(model.get("camera_id")),
        "checkpoint_id": _json_value(model.get("checkpoint_id")),
        "checkpoint_sha256": _json_value(model.get("checkpoint_sha256")),
        "decision_threshold": _number(model.get("decision_threshold")),
        "score_semantics": _json_value(model.get("score_semantics")),
        "validated_on_real_csi": False,
        "message": None,
        "test_metrics": _json_value(model.get("test_metrics")),
        "clip_level_metrics": _json_value(model.get("clip_level_metrics")),
        "examples": exported_examples,
    }


def _prediction_payload(
    prediction: dict[str, Any] | None, evidence: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    delta_e = _number(prediction.get("delta_e_2000"))
    return {
        "observed_rgb": _rgb(prediction.get("observed_rgb")),
        "observed_lab": [_number(value) for value in prediction.get("observed_lab") or []]
        or None,
        "expected_rgb": _rgb(prediction.get("expected_rgb")),
        "expected_lab": [_number(value) for value in prediction.get("expected_lab") or []]
        or None,
        "expected_color_source": _json_value(prediction.get("expected_color_source")),
        "delta_e_2000": delta_e,
        "mismatch_score": _number(prediction.get("mismatch_score")),
        "measurement_confidence": _number(prediction.get("measurement_confidence")),
        "alignment_confidence": _number(prediction.get("alignment_confidence")),
        "evidence_image_paths": [
            item["frame_url"] for item in evidence if item.get("frame_url") is not None
        ],
        "explanation": _json_value(prediction.get("explanation")),
        "verification_status": _json_value(prediction.get("verification_status"))
        or "unverified",
        "ground_truth_anomaly": _json_value(prediction.get("ground_truth_anomaly")),
        "review_status": "discrepancy_measured" if delta_e is not None else "not_available",
        "review_basis": "descriptive_delta_e_only",
    }


def _select_evidence(frame_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = frame_rows.copy()
    stage_rank = {stage: index for index, stage in enumerate(PREFERRED_EVIDENCE_STAGES)}
    ranked["_stage_rank"] = ranked["stage"].map(stage_rank).fillna(len(stage_rank))
    ranked["_confidence_rank"] = pd.to_numeric(ranked.get("confidence"), errors="coerce").fillna(-1)
    return ranked.sort_values(
        ["_stage_rank", "_confidence_rank", "corrected_media_seconds"],
        ascending=[True, False, True],
    ).head(2)


def export_dashboard(
    output_root: Path,
    frontend_public_root: Path,
    *,
    per_frame_path: Path | None = None,
    per_vial_path: Path | None = None,
    predictions_root: Path | None = None,
    ml_predictions_path: Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Export existing pipeline artifacts without rerunning measurement or prediction logic."""
    output_root = output_root.resolve()
    frontend_public_root = frontend_public_root.resolve()
    per_frame_path = (
        per_frame_path or output_root / "data" / "processed" / "per_frame.csv"
    ).resolve()
    per_vial_path = (per_vial_path or output_root / "data" / "processed" / "per_vial.csv").resolve()
    predictions_root = (
        predictions_root or output_root / "artifacts" / "predictions"
    ).resolve()
    ml_predictions_path = (
        ml_predictions_path
        or output_root / "artifacts" / "ml_predictions" / "causal_tcn_examples.json"
    ).resolve()
    for path in (per_frame_path, per_vial_path):
        if not path.is_file():
            raise FileNotFoundError(f"required dashboard input does not exist: {path}")

    frames = pd.read_csv(per_frame_path)
    vials = pd.read_csv(per_vial_path).sort_values("vial_id").reset_index(drop=True)
    required_frame_columns = {
        "vial_id",
        "stage",
        "status",
        "confidence",
        "corrected_media_seconds",
        "image_path",
        "color_qa_path",
    }
    required_vial_columns = {"vial_id", "video_id", "camera_id", "recipe_id"}
    if missing := required_frame_columns.difference(frames.columns):
        raise ValueError(f"per-frame table is missing columns: {', '.join(sorted(missing))}")
    if missing := required_vial_columns.difference(vials.columns):
        raise ValueError(f"per-vial table is missing columns: {', '.join(sorted(missing))}")

    generated_root = frontend_public_root / "generated"
    image_root = generated_root / "images"
    predictions = _load_predictions(predictions_root)
    ml_prediction = _load_ml_prediction(
        ml_predictions_path, output_root, image_root
    )
    inspections: list[dict[str, Any]] = []
    previous_recipe_id: int | None = None

    for vial in vials.to_dict(orient="records"):
        vial_id = int(vial["vial_id"])
        recipe_id = int(vial["recipe_id"])
        vial_frames = frames.loc[pd.to_numeric(frames["vial_id"]) == vial_id].copy()
        evidence: list[dict[str, Any]] = []
        for frame in _select_evidence(vial_frames).to_dict(orient="records"):
            stage = str(frame["stage"])
            frame_url = _copy_browser_asset(
                _relative_source_path(output_root, frame.get("image_path")),
                image_root,
                prefix="frame",
                vial_id=vial_id,
                stage=stage,
            )
            qa_url = _copy_browser_asset(
                _relative_source_path(output_root, frame.get("color_qa_path")),
                image_root,
                prefix="qa",
                vial_id=vial_id,
                stage=stage,
            )
            evidence.append(
                {
                    "stage": stage,
                    "status": str(frame.get("status") or "unknown"),
                    "confidence": _number(frame.get("confidence")),
                    "observed_rgb": _rgb(frame.get("observed_rgb")),
                    "observed_lab": [
                        _number(frame.get("lab_l_median")),
                        _number(frame.get("lab_a_median")),
                        _number(frame.get("lab_b_median")),
                    ],
                    "decoded_frame_index": _number(
                        frame.get("decoded_frame_index"), integer=True
                    ),
                    "media_seconds": _number(frame.get("corrected_media_seconds")),
                    "frame_url": frame_url,
                    "qa_url": qa_url,
                    "image_available": frame_url is not None,
                    "qa_available": qa_url is not None,
                }
            )

        statuses = vial_frames["status"].fillna("unknown").astype(str)
        actual_previous = previous_recipe_id
        is_transition = actual_previous is not None and recipe_id != actual_previous
        prediction = _prediction_payload(predictions.get(vial_id), evidence)
        recipe_source = vial_frames.iloc[0].to_dict() if not vial_frames.empty else vial
        inspections.append(
            {
                "vial_id": vial_id,
                "video_id": int(vial["video_id"]),
                "camera_id": str(vial["camera_id"]),
                "recipe_id": recipe_id,
                "previous_recipe_id": actual_previous,
                "is_recipe_transition": is_transition,
                "measurement": {
                    "status": "low_confidence"
                    if statuses.eq("low_confidence").any()
                    else ("ok" if statuses.eq("ok").any() else "missing"),
                    "frame_count": int(len(vial_frames)),
                    "measured_frame_count": int(statuses.isin(["ok", "low_confidence"]).sum()),
                    "ok_frame_count": int(statuses.eq("ok").sum()),
                    "low_confidence_frame_count": int(statuses.eq("low_confidence").sum()),
                    "missing_frame_count": int(statuses.isin(["missing", "not_measured"]).sum()),
                    "confidence_median": _number(vial.get("confidence_median")),
                    "alignment_confidence": _number(vial.get("alignment_confidence")),
                    "observed_lab_median": [
                        _number(vial.get("lab_l_median_median")),
                        _number(vial.get("lab_a_median_median")),
                        _number(vial.get("lab_b_median_median")),
                    ],
                },
                "recipe": {
                    "type": _json_value(recipe_source.get("type")),
                    "t100_ml": _number(vial.get("t100_ml")),
                    "t200_ml": _number(vial.get("t200_ml")),
                    "t400_ml": _number(vial.get("t400_ml")),
                    "total_ml": _number(vial.get("total_ml")),
                    "dispense_order": _json_value(vial.get("dispense_order")),
                    "color_purpose": _json_value(recipe_source.get("color_purpose")),
                },
                "prediction": prediction,
                "evidence": evidence,
                "label_status": _json_value(vial.get("label_status")) or "unverified",
                "ground_truth_anomaly": _json_value(vial.get("ground_truth_anomaly")),
            }
        )
        previous_recipe_id = recipe_id

    transitions = [
        {
            "vial_id": item["vial_id"],
            "from_recipe_id": item["previous_recipe_id"],
            "to_recipe_id": item["recipe_id"],
            "prediction": item["prediction"],
            "evidence": item["evidence"],
            "ground_truth_anomaly": item["ground_truth_anomaly"],
        }
        for item in inspections
        if item["is_recipe_transition"]
    ]
    status_values = frames["status"].fillna("unknown").astype(str)
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "provenance": {
            "source": "mixing_copilot_opencv",
            "integration_mode": "static_artifact_export",
            "per_frame_path": str(per_frame_path),
            "per_frame_sha256": sha256_file(per_frame_path),
            "per_vial_path": str(per_vial_path),
            "per_vial_sha256": sha256_file(per_vial_path),
            "prediction_files_found": len(predictions),
            "ml_prediction_artifact": (
                str(ml_predictions_path) if ml_predictions_path.is_file() else None
            ),
            "ml_prediction_artifact_sha256": (
                sha256_file(ml_predictions_path) if ml_predictions_path.is_file() else None
            ),
            "measurement_semantics": "OpenCV pixel measurements",
            "discrepancy_semantics": "descriptive rule-based comparison, not contamination",
            "mismatch_score_semantics": "normalized mismatch index, not probability",
            "expected_color_semantics": "unvalidated recipe-color proxy",
        },
        "summary": {
            "inspected_vials": len(inspections),
            "frame_records": len(frames),
            "measured_frames": int(status_values.isin(["ok", "low_confidence"]).sum()),
            "ok_frames": int(status_values.eq("ok").sum()),
            "low_confidence_frames": int(status_values.eq("low_confidence").sum()),
            "missing_or_not_measured_frames": int(
                status_values.isin(["missing", "not_measured"]).sum()
            ),
            "predictions_available": sum(item["prediction"] is not None for item in inspections),
            "discrepancy_reviews": sum(
                item["prediction"] is not None
                and item["prediction"]["delta_e_2000"] is not None
                for item in inspections
            ),
            "recipe_transitions": len(transitions),
        },
        "inspections": inspections,
        "transitions": transitions,
        "ml_prediction": ml_prediction,
    }
    destination = generated_root / "mixing_copilot.json"
    atomic_json_dump(payload, destination)
    return payload | {"output_path": str(destination), "image_root": str(image_root)}
