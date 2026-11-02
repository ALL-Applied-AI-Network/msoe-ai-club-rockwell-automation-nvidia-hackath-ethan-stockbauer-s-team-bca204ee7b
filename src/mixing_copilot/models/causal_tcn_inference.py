from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Optional dependency.
    torch = None
    nn = None


MODEL_VERSION = "ethan_causal_tcn_v1"
IMAGE_FEATURE_SIZE = 19
HISTORY_FRAMES = 32
EXPECTED_CAMERA_ID = "FillCam"
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
TRAINING_DOMAIN = "synthetic_isaac_sim_recipe_5_6"


if nn is not None:

    class CheckpointCompatibleCausalTCN(nn.Module):
        """Exact architecture used by Ethan's refined Video 5 checkpoint."""

        def __init__(self, input_size: int = IMAGE_FEATURE_SIZE, channels: int = 32):
            super().__init__()
            self.input = nn.Conv1d(input_size, channels, 1)
            self.blocks = nn.ModuleList(
                [
                    nn.Conv1d(channels, channels, 3, dilation=dilation)
                    for dilation in (1, 2, 4, 8)
                ]
            )
            self.output = nn.Linear(channels, 1)

        def forward(self, inputs):
            values = self.input(inputs)
            for block in self.blocks:
                padding = 2 * block.dilation[0]
                values = nn.functional.pad(values, (padding, 0))
                values = torch.relu(block(values))
            return self.output(values[:, :, -1]).squeeze(1)

else:

    class CheckpointCompatibleCausalTCN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required; install the project with the [tcn] extra")


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required; install the project with the [tcn] extra")


def _extended_path(path: Path) -> str:
    absolute = str(path.expanduser().resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return f"\\\\?\\{absolute}"
    return absolute


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_extended_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frame(path: Path) -> np.ndarray:
    resolved = Path(path)
    try:
        with open(_extended_path(resolved), "rb") as handle:
            prefix = handle.read(64)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"frame does not exist: {path}") from error
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"frame is a Git LFS pointer, not a decoded image: {path}")
    encoded = np.fromfile(_extended_path(resolved), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode frame: {path}")
    return image


def frame_features_from_image(
    image: np.ndarray,
    *,
    camera_id: str = EXPECTED_CAMERA_ID,
    source: str = "<image>",
) -> np.ndarray:
    """Extract Ethan's exact 19 calibrated ROI features from one FillCam frame."""
    if camera_id != EXPECTED_CAMERA_ID:
        raise ValueError(
            f"checkpoint is calibrated for {EXPECTED_CAMERA_ID}; received {camera_id}"
        )
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a three-channel BGR frame: {source}")
    height, width = image.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(
            f"checkpoint expects {EXPECTED_WIDTH}x{EXPECTED_HEIGHT} FillCam frames; "
            f"received {width}x{height}: {source}"
        )
    crop = image[
        round(height * 366 / 480) : round(height * 407 / 480),
        round(width * 294 / 640) : round(width * 355 / 640),
    ]
    if crop.size == 0:
        raise ValueError(f"frame is too small for the calibrated liquid ROI: {source}")
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_normalized = hsv / np.array([180.0, 255.0, 255.0], dtype=np.float32)
    colored = hsv[:, :, 1] > 35
    colored_rgb = rgb[colored]
    colored_hsv = hsv_normalized[colored]
    if not len(colored_rgb):
        colored_rgb = np.zeros((1, 3), dtype=np.float32)
        colored_hsv = np.zeros((1, 3), dtype=np.float32)
    return np.array(
        [
            *rgb.mean(axis=(0, 1)),
            *rgb.std(axis=(0, 1)),
            *hsv_normalized.mean(axis=(0, 1)),
            *hsv_normalized.std(axis=(0, 1)),
            float(colored.mean()),
            *colored_rgb.mean(axis=0),
            *colored_hsv.mean(axis=0),
        ],
        dtype=np.float32,
    )


def frame_features(path: Path, *, camera_id: str = EXPECTED_CAMERA_ID) -> np.ndarray:
    return frame_features_from_image(
        _read_frame(path), camera_id=camera_id, source=str(path)
    )


def _safe_checkpoint_load(path: Path) -> dict[str, Any]:
    _require_torch()
    safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            _extended_path(path), map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    missing = {"model", "config", "feature_mean", "feature_std"}.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(sorted(missing))}")
    return checkpoint


class CausalTCNPredictor:
    """Load the immutable trained checkpoint and score causal FillCam histories."""

    def __init__(self, checkpoint_path: Path):
        _require_torch()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = _safe_checkpoint_load(self.checkpoint_path)
        self.config = dict(checkpoint["config"])
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        if self.feature_mean.shape != (IMAGE_FEATURE_SIZE,) or self.feature_std.shape != (
            IMAGE_FEATURE_SIZE,
        ):
            raise ValueError("checkpoint normalization is not compatible with 19 image features")
        if np.any(self.feature_std <= 0):
            raise ValueError("checkpoint feature standard deviations must be positive")
        self.target = str(self.config.get("target"))
        if self.target != "target_future_reject":
            raise ValueError(f"unsupported checkpoint target: {self.target}")
        self.threshold = float(self.config["decision_threshold"])
        self.positive_class_weight = float(self.config["positive_class_weight"])
        if not 0 <= self.threshold <= 1 or self.positive_class_weight <= 0:
            raise ValueError("checkpoint threshold or class weight is invalid")
        self.model = CheckpointCompatibleCausalTCN(input_size=IMAGE_FEATURE_SIZE)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)

    def predict(
        self,
        frame_paths: Sequence[Path],
        *,
        camera_id: str = EXPECTED_CAMERA_ID,
        timestamps_s: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        if not frame_paths:
            raise ValueError("at least one frame is required")
        if timestamps_s is not None:
            if len(timestamps_s) != len(frame_paths):
                raise ValueError("timestamps must have the same length as frame paths")
            if any(
                current < previous
                for previous, current in zip(
                    timestamps_s, timestamps_s[1:], strict=False
                )
            ):
                raise ValueError("timestamps must be ordered from past to current")
        selected = [Path(path) for path in frame_paths[-HISTORY_FRAMES:]]
        values = [frame_features(path, camera_id=camera_id) for path in selected]
        if len(values) < HISTORY_FRAMES:
            values = [values[0]] * (HISTORY_FRAMES - len(values)) + values
        sequence = np.stack(values, axis=1)
        normalized = (
            sequence - self.feature_mean[:, None]
        ) / self.feature_std[:, None]
        inputs = torch.from_numpy(normalized.astype(np.float32))[None]
        if tuple(inputs.shape) != (1, IMAGE_FEATURE_SIZE, HISTORY_FRAMES):
            raise RuntimeError(f"unexpected model input shape: {tuple(inputs.shape)}")
        with torch.no_grad():
            raw_logit = float(self.model(inputs).item())
            corrected_logit = raw_logit - math.log(self.positive_class_weight)
            score = float(torch.sigmoid(torch.tensor(corrected_logit)).item())
        return {
            "raw_model_logit": raw_logit,
            "score": score,
            "threshold": self.threshold,
            "decision": "REJECT" if score >= self.threshold else "PASS",
            "target": self.target,
            "input_shape": [1, IMAGE_FEATURE_SIZE, HISTORY_FRAMES],
            "frames_supplied": len(frame_paths),
            "frames_used": len(selected),
        }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _demo_records(
    records: list[dict[str, Any]], target: str, split: str
) -> list[dict[str, Any]]:
    split_records = [record for record in records if record.get("split") == split]
    reject_candidates = [record for record in split_records if int(record.get(target, 0)) == 1]
    if not reject_candidates:
        raise ValueError(f"split {split!r} has no positive {target} records")
    reject = max(reject_candidates, key=lambda record: int(record["frame_index"]))
    pass_candidates = [
        record
        for record in split_records
        if float(record.get("contamination_ml", 0)) == 0
        and int(record["frame_index"]) == int(reject["frame_index"])
        and int(record["recipe_before"]) == int(reject["recipe_before"])
        and int(record["recipe_after"]) == int(reject["recipe_after"])
    ]
    if not pass_candidates:
        raise ValueError("no matched synthetic PASS record exists for the selected REJECT record")
    return [pass_candidates[0], reject]


def _manifest_horizon_seconds(records: list[dict[str, Any]], target: str) -> float | None:
    by_clip: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        by_clip.setdefault(str(record["clip_id"]), {})[int(record["frame_index"])] = record
    leads: list[float] = []
    for record in records:
        if int(record.get(target, 0)) != 1:
            continue
        onset = int(record["contamination_onset_frame"])
        onset_record = by_clip[str(record["clip_id"])].get(onset)
        if onset_record is not None:
            leads.append(float(onset_record["timestamp_s"]) - float(record["timestamp_s"]))
    return max(leads) if leads else None


def generate_demo_predictions(
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    camera_id: str = EXPECTED_CAMERA_ID,
    split: str = "test",
) -> dict[str, Any]:
    """Generate one genuine matched synthetic PASS/REJECT dashboard pair."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    records = _load_manifest(manifest_path)
    predictor = CausalTCNPredictor(checkpoint_path)
    selected = _demo_records(records, predictor.target, split)
    examples: list[dict[str, Any]] = []
    for record in selected:
        paths = [manifest_path.parent / relative for relative in record["input_frames"]]
        prediction = predictor.predict(paths, camera_id=camera_id)
        contamination_ml = float(record.get("contamination_ml", 0))
        actual_clip_label = "REJECT" if contamination_ml > 0 else "PASS"
        actual_target_label = "REJECT" if int(record[predictor.target]) else "PASS"
        frame_index = int(record["frame_index"])
        onset = int(record["contamination_onset_frame"])
        examples.append(
            {
                "example_id": f"{split}-{actual_clip_label.lower()}-frame-{frame_index}",
                "source_clip": str(record["clip_id"]),
                "source_split": split,
                "camera_id": camera_id,
                "observation_frame": frame_index,
                "observation_timestamp_s": float(record["timestamp_s"]),
                "observation_stage": (
                    "before_visible_contamination" if frame_index < onset else "visible_fill"
                ),
                "window_start_frame": int(Path(record["input_frames"][0]).stem.split("_")[-1]),
                "window_end_frame": int(Path(record["input_frames"][-1]).stem.split("_")[-1]),
                "actual_synthetic_label": actual_clip_label,
                "actual_target_label": actual_target_label,
                "contamination_ml": contamination_ml,
                "evidence_image_path": str(paths[-1].resolve()),
                **prediction,
            }
        )
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    horizon = _manifest_horizon_seconds(records, predictor.target)
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": timestamp,
        "model": {
            "model_name": "Causal TCN",
            "model_type": "causal_tcn",
            "model_version": MODEL_VERSION,
            "purpose": "Predict a future synthetic rejection from early-fill visual observations.",
            "training_domain": TRAINING_DOMAIN,
            "target": predictor.target,
            "horizon_seconds": horizon,
            "input_shape": [1, IMAGE_FEATURE_SIZE, HISTORY_FRAMES],
            "feature_count": IMAGE_FEATURE_SIZE,
            "history_frames": HISTORY_FRAMES,
            "camera_id": EXPECTED_CAMERA_ID,
            "checkpoint_id": predictor.checkpoint_path.name,
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "decision_threshold": predictor.threshold,
            "score_semantics": (
                "Class-weight-corrected sigmoid model score; not a calibrated physical "
                "contamination probability."
            ),
            "validated_on_real_csi": False,
            "test_metrics": predictor.config.get("test"),
            "clip_level_metrics": predictor.config.get(
                "clip_level_first_2s_after_visible_onset"
            ),
        },
        "examples": examples,
    }
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload | {"output_path": str(output_path)}
