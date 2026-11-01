import json

from mixing_copilot.prediction import VialPrediction


def test_prediction_serializes_nullable_verification_fields():
    prediction = VialPrediction(
        vial_id=1,
        recipe_id=5,
        observed_rgb=None,
        observed_lab=None,
        expected_rgb=None,
        expected_lab=None,
        expected_color_source=None,
        delta_e_2000=None,
        mismatch_score=None,
        measurement_confidence=0,
        alignment_confidence=None,
        evidence_image_paths=[],
        explanation="Insufficient evidence.",
    )
    payload = json.loads(prediction.model_dump_json())
    assert payload["ground_truth_anomaly"] is None
    assert payload["verification_status"] == "unverified"
