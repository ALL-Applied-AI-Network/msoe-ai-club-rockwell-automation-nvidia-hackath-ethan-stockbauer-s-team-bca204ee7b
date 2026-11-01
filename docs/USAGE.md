# Mixing Copilot usage

## Setup on Windows PowerShell

This checkout currently includes a workspace-local Python 3.11 under `.tools/python311`. That folder
is ignored by Git. On another machine, install Python 3.11 and create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Update `dataset_root` in `configs/dataset.yaml`. The source dataset is always treated as read-only.

## Commands

```powershell
mixing-copilot audit --config configs/dataset.yaml
mixing-copilot align --config configs/dataset.yaml
mixing-copilot extract --config configs/dataset.yaml --video-id 5
mixing-copilot features --config configs/features.yaml
mixing-copilot build-dataset --config configs/dataset.yaml
mixing-copilot predict --config configs/dataset.yaml --vial-id 30
mixing-copilot run --config configs/dataset.yaml
```

With the workspace-local interpreter, replace `mixing-copilot` with:

```powershell
.\.tools\python311\python.exe -m mixing_copilot.cli
```

Use `--dry-run` with `audit`, `align`, or `extract` to inspect planned inputs. Use
`run --skip-audit` after the source manifest has already been generated.

## Verification

```powershell
python -m pytest
python -m ruff check .
```

The color ROI is calibrated only for the fixed 640x480 `vial_fill` view. Inspect
`artifacts/qa/color` after camera movement or lighting changes. A low-confidence measurement must not
be silently converted into a high-confidence risk score.

## Interpretation

`mismatch_score` is a normalized Delta E color mismatch index. It is not a probability of
contamination. Expected colors currently come from the digital twin's uncalibrated tank-setpoint
weighted sRGB proxy. `ground_truth_anomaly` remains null and `verification_status` remains
`unverified`.

## Processing an Isaac Sim synthetic recipe video

The synthetic adapter accepts the three files produced by `render_recipe_video.py`: an MP4,
`*_frame_labels.csv`, and `*_recipes.csv`. It validates the video/CSV contract before extracting
frames. Synthetic frame indices and `video_s` values are authoritative, so this path deliberately
does not run the CSI still-image timestamp alignment.

Validate a dataset without extracting frames:

```powershell
.\.tools\python311\python.exe -m mixing_copilot.cli process-synthetic `
  --video C:\path\to\video5_synthetic.mp4 `
  --frame-labels C:\path\to\video5_synthetic_frame_labels.csv `
  --recipes C:\path\to\video5_synthetic_recipes.csv `
  --dataset-id video5_synthetic `
  --video-id 5 `
  --camera-id FillCam `
  --validate-only
```

Run validation, direct frame extraction, OpenCV measurement, and dataset building:

```powershell
.\.tools\python311\python.exe -m mixing_copilot.cli process-synthetic `
  --video C:\path\to\video5_synthetic.mp4 `
  --frame-labels C:\path\to\video5_synthetic_frame_labels.csv `
  --recipes C:\path\to\video5_synthetic_recipes.csv `
  --dataset-id video5_synthetic `
  --video-id 5 `
  --camera-id FillCam `
  --features-config configs\features.yaml
```

Outputs are isolated from the CSI Video 5 outputs:

- validation: `reports/synthetic/<dataset-id>_validation.json`;
- extracted frames and feature table: `data/interim/synthetic/<dataset-id>/`;
- per-frame and per-vial datasets: `data/processed/synthetic/<dataset-id>/`;
- color QA overlays: `artifacts/qa/synthetic/<dataset-id>/color/`.

The adapter verifies contiguous MP4 frame indices, timestamp/FPS consistency, non-overlapping vial
boundaries, recipe volumes, and matching per-frame metadata. It assigns
`alignment_method=synthetic_direct_frame_index`, leaves `ground_truth_anomaly` null, and marks labels
`synthetic_unverified`. It does not train or run the Causal TCN.

`--camera-id` defaults to `FillCam`; this is an explicit assumption, not camera autodetection.
`CapSealCam` inputs may be checked with `--validate-only`, but color extraction is rejected until a
camera-specific ROI is calibrated. `--video-id` remains an integer for compatibility with existing
processed-data schemas, while `--dataset-id` provides the synthetic run identifier.

Between-fill rows remain in the manifest as `pre_fill` or `post_fill` transition evidence, with
`measurement_eligible=false`. The feature table marks them `not_measured`, does not create color QA
overlays for them, and excludes them from per-vial color aggregation. They are never treated as
recipe-associated liquid measurements.

Isaac Sim's `FillCam` currently renders 640x480, matching the existing feature configuration, but
the liquid ROI must still be inspected in the generated QA overlays. Supply a separately calibrated
feature configuration if camera placement, resolution, or framing changes. Validation-only mode
does not require a feature configuration.
