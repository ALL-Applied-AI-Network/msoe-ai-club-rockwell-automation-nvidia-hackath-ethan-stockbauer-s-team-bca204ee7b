from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixing_copilot.models.causal_tcn_inference import (  # noqa: E402
    CausalTCNPredictor,
    CheckpointCompatibleCausalTCN,
    frame_features_from_image,
    generate_demo_predictions,
)


def _frame(path, color=(20, 80, 180)):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[366:407, 294:355] = color
    assert cv2.imwrite(str(path), image)
    return path


def _checkpoint(path):
    model = CheckpointCompatibleCausalTCN(input_size=19)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "target": "target_future_reject",
                "decision_threshold": 0.5,
                "positive_class_weight": 1.0,
                "test": {"roc_auc": 0.75},
            },
            "feature_mean": np.zeros(19, dtype=np.float32),
            "feature_std": np.ones(19, dtype=np.float32),
        },
        path,
    )
    return path


def test_feature_extractor_validates_camera_and_dimensions():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert frame_features_from_image(image).shape == (19,)
    with pytest.raises(ValueError, match="FillCam"):
        frame_features_from_image(image, camera_id="CapSealCam")
    with pytest.raises(ValueError, match="640x480"):
        frame_features_from_image(np.zeros((240, 320, 3), dtype=np.uint8))


def test_checkpoint_predictor_preserves_shape_threshold_and_causality(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    frames = [_frame(tmp_path / f"frame_{index:06d}.png") for index in range(2)]
    predictor = CausalTCNPredictor(checkpoint)

    result = predictor.predict(frames, timestamps_s=[0.0, 0.1])

    assert result["input_shape"] == [1, 19, 32]
    assert result["frames_supplied"] == 2
    assert result["frames_used"] == 2
    assert result["score"] == pytest.approx(0.5)
    assert result["threshold"] == 0.5
    assert result["decision"] == "REJECT"
    with pytest.raises(ValueError, match="ordered"):
        predictor.predict(frames, timestamps_s=[0.1, 0.0])


def test_predictor_rejects_git_lfs_pointer(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    pointer = tmp_path / "frame_000000.png"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        CausalTCNPredictor(checkpoint).predict([pointer])


def test_generate_demo_predictions_keeps_synthetic_labels_separate(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    frames = [_frame(tmp_path / f"frame_{index:06d}.png") for index in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for contamination, label, clip in ((0.0, 0, "pass"), (8.0, 1, "reject")):
        rows.append(
            {
                "clip_id": clip,
                "split": "test",
                "frame_index": 1,
                "timestamp_s": 0.1,
                "input_frames": [frame.name for frame in frames],
                "target_future_reject": label,
                "contamination_ml": contamination,
                "contamination_onset_frame": 2,
                "recipe_before": 5,
                "recipe_after": 6,
            }
        )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "prediction.json"

    result = generate_demo_predictions(checkpoint, manifest, output)

    assert [item["actual_synthetic_label"] for item in result["examples"]] == [
        "PASS",
        "REJECT",
    ]
    assert all(item["source_split"] == "test" for item in result["examples"])
    assert result["model"]["validated_on_real_csi"] is False
    assert output.is_file()
