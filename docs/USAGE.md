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
