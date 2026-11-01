import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from mixing_copilot.dashboard import export_dashboard


def _write_fixture(tmp_path):
    output_root = tmp_path / "output"
    public_root = tmp_path / "frontend" / "public"
    frame_root = output_root / "frames"
    qa_root = output_root / "qa"
    frame_root.mkdir(parents=True)
    qa_root.mkdir(parents=True)
    (frame_root / "vial1_late.jpg").write_bytes(b"frame-one")
    (qa_root / "vial1_late.jpg").write_bytes(b"qa-one")

    frames = pd.DataFrame(
        [
            {
                "vial_id": 2,
                "stage": "late_fill",
                "status": "low_confidence",
                "confidence": 0.2,
                "corrected_media_seconds": 2.0,
                "decoded_frame_index": 60,
                "image_path": "frames/missing.jpg",
                "color_qa_path": "qa/missing.jpg",
                "observed_rgb": "[8, 9, 10]",
                "lab_l_median": 20,
                "lab_a_median": 1,
                "lab_b_median": 2,
                "type": "Double",
                "color_purpose": "fixture two",
            },
            {
                "vial_id": 1,
                "stage": "late_fill",
                "status": "ok",
                "confidence": 0.9,
                "corrected_media_seconds": 1.0,
                "decoded_frame_index": 30,
                "image_path": "frames/vial1_late.jpg",
                "color_qa_path": "qa/vial1_late.jpg",
                "observed_rgb": "[1, 2, 3]",
                "lab_l_median": 10,
                "lab_a_median": -1,
                "lab_b_median": 4,
                "type": "Triple",
                "color_purpose": "fixture one",
            },
        ]
    )
    vials = pd.DataFrame(
        [
            {
                "vial_id": 2,
                "video_id": 5,
                "camera_id": "vial_fill",
                "recipe_id": 6,
                "alignment_confidence": 0.7,
                "confidence_median": 0.2,
                "lab_l_median_median": 20,
                "lab_a_median_median": 1,
                "lab_b_median_median": 2,
                "t100_ml": 10,
                "t200_ml": 10,
                "t400_ml": 0,
                "total_ml": 20,
                "dispense_order": "T100>T200",
                "label_status": "unverified",
                "ground_truth_anomaly": None,
            },
            {
                "vial_id": 1,
                "video_id": 5,
                "camera_id": "vial_fill",
                "recipe_id": 5,
                "alignment_confidence": 0.8,
                "confidence_median": 0.9,
                "lab_l_median_median": 10,
                "lab_a_median_median": -1,
                "lab_b_median_median": 4,
                "t100_ml": 10,
                "t200_ml": 10,
                "t400_ml": 10,
                "total_ml": 30,
                "dispense_order": "T100>T200>T400",
                "label_status": "unverified",
                "ground_truth_anomaly": None,
            },
        ]
    )
    processed = output_root / "data" / "processed"
    processed.mkdir(parents=True)
    frames.to_csv(processed / "per_frame.csv", index=False)
    vials.to_csv(processed / "per_vial.csv", index=False)

    predictions = output_root / "artifacts" / "predictions"
    predictions.mkdir(parents=True)
    (predictions / "vial_01.json").write_text(
        json.dumps(
            {
                "vial_id": 1,
                "recipe_id": 5,
                "observed_rgb": [1, 2, 3],
                "observed_lab": [10, -1, 4],
                "expected_rgb": [4, 5, 6],
                "expected_lab": [11, 0, 5],
                "expected_color_source": "synthetic_tank_setpoint_weighted_srgb_unvalidated",
                "delta_e_2000": 7.5,
                "mismatch_score": 0.15,
                "measurement_confidence": 0.9,
                "alignment_confidence": 0.8,
                "evidence_image_paths": ["C:/not/browser/safe.jpg"],
                "explanation": "Descriptive mismatch only.",
                "verification_status": "unverified",
                "ground_truth_anomaly": None,
            }
        ),
        encoding="utf-8",
    )
    return output_root, public_root


def test_dashboard_export_is_ordered_versioned_and_browser_safe(tmp_path):
    output_root, public_root = _write_fixture(tmp_path)
    generated_at = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)

    result = export_dashboard(output_root, public_root, generated_at=generated_at)

    assert result["schema_version"] == "1.0"
    assert result["generated_at_utc"] == "2026-11-01T12:30:00Z"
    assert [item["vial_id"] for item in result["inspections"]] == [1, 2]
    assert result["summary"] == {
        "inspected_vials": 2,
        "frame_records": 2,
        "measured_frames": 2,
        "ok_frames": 1,
        "low_confidence_frames": 1,
        "missing_or_not_measured_frames": 0,
        "predictions_available": 1,
        "discrepancy_reviews": 1,
        "recipe_transitions": 1,
    }
    first = result["inspections"][0]
    assert first["prediction"]["review_basis"] == "descriptive_delta_e_only"
    assert first["prediction"]["ground_truth_anomaly"] is None
    assert first["prediction"]["evidence_image_paths"] == [
        "/generated/images/frame_vial0001_late_fill.jpg"
    ]
    assert first["evidence"][0]["qa_url"] == "/generated/images/qa_vial0001_late_fill.jpg"
    assert (public_root / "generated" / "images" / "frame_vial0001_late_fill.jpg").is_file()


def test_dashboard_export_preserves_missing_predictions_and_images(tmp_path):
    output_root, public_root = _write_fixture(tmp_path)

    result = export_dashboard(output_root, public_root)

    second = result["inspections"][1]
    assert second["prediction"] is None
    assert second["ground_truth_anomaly"] is None
    assert second["measurement"]["status"] == "low_confidence"
    assert second["evidence"][0]["frame_url"] is None
    assert second["evidence"][0]["qa_url"] is None
    serialized = json.loads(
        (public_root / "generated" / "mixing_copilot.json").read_text(encoding="utf-8")
    )
    assert "output_path" not in serialized
    assert serialized["inspections"][1]["prediction"] is None


def test_dashboard_export_requires_processed_tables(tmp_path):
    with pytest.raises(FileNotFoundError, match="required dashboard input"):
        export_dashboard(tmp_path / "missing", tmp_path / "public")
