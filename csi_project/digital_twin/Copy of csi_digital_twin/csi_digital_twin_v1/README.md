# CSI vial line: digital twin

This is a digital twin of the CSI vial line, prepared for generating synthetic data in NVIDIA Isaac Sim / Omniverse. It contains two stations, each with a camera matched to a real camera on the line:

- **Tray inspection station:** robot, inspection trays and capped vials.
- **Fill (nozzle) station:** vial carts under the dispensing nozzle, with a filling animation.

Units are **meters** and **Z is up**. There is **no physics** anywhere: everything is static geometry, except the fill station's pre-recorded filling animation.

## Contents
```
csi_line.usda                        the full line (open this for everything)
stations/
  tray_inspection/tray_inspection.usdc   tray inspection station + cap-seal camera + its lights
  fill_station/fill_station.usda         fill station + fill camera + its lights (animated, frames 1-686)
  magnemover/magnemover.usdc             MagneMover track and carts
  facility/facility.usdc                 the rest of the plant, floor and dome light
cameras/                             camera settings (JSON), one file per camera
tools/
  render_example.py                  render one image from any camera (Isaac Sim)
  apply_lens.py                      adds the real cap-seal camera lens distortion to a render
```
Keep the folder structure as it is: the files reference each other by relative paths.

## The full line or a single station?
`csi_line.usda` **is** the digital twin. It does not hold any geometry itself: it references the four station files and places them together.

| You want to... | Open |
|---|---|
| See or render the whole line | `csi_line.usda` |
| Work only on the tray inspection data | `stations/tray_inspection/tray_inspection.usdc` |
| Work only on the fill station data | `stations/fill_station/fill_station.usda` |

- **Open a station file on its own** when you only need that station. It loads in seconds, while the full line loads the whole plant (about 0.7 GB). Each station file has its own camera(s) and lights, so you can render from it directly.
- **To change something inside a station** (a material, a vial, a light), **edit the station file.** The full line picks up the change automatically.
- **Edit `csi_line.usda` only for line-level changes:** where the stations sit relative to each other, or switching parts of the line on or off.

  If you edit a station's contents while `csi_line.usda` is open, Isaac Sim saves the change into `csi_line.usda` as an override. It applies only to the full line, not to the station file.
- The tray inspection, MagneMover and facility files use the same coordinates as the full line. The fill station file uses its own local coordinates, and `csi_line.usda` moves it into place on the track.

## Cameras
| Camera | Station | Final image | Prim path (in the station file) |
|---|---|---|---|
| `CapSealCam` | tray inspection | 640 × 480 | `/TrayInspection/CapSealCam/CapSealCam` |
| `FillCam` | fill station | 640 × 480 | `/FillStation/FillStation_LocalRoot/FillCam/FillCam` |

- **Paths in the full line:** add the prefix `/CSI_Line`. For example, `/CSI_Line/TrayInspection/CapSealCam/CapSealCam`.
- **Camera JSON files:** `cameras/<name>.json` has each camera's position, orientation, focal length, resolution and lens values, plus its prim path in both the station file and the full line.
- **Cap-seal camera:** `CapSealCam` matches the real 640 × 480 cap-seal camera, solved from the cap_seal video clip.
- **Lens distortion (CapSealCam):** the real camera has a slight lens distortion, so:
  1. Render at `render_px` (768 × 576, a 1.2× wider pinhole view).
  2. Run `tools/apply_lens.py`, which adds the distortion and crops to 640 × 480.
- **Fill camera:** a plain pinhole. Render 640 × 480 directly.

## Fill station animation
- The fill station plays a filling sequence over time codes **1–686** at **30 fps** (22.9 s). Blue liquid fills, then clear liquid is added, while the next cart moves in.
- Pick the moment by time code. For example, 301 is mid-fill and 686 is the finished vial.
- Everything else in the line is static.

## Recipe-conditioned synthetic renders
`tools/render_batch1.py` reads vial recipe metadata and samples valid fill-animation time codes; it does not use the source-video frame numbers. It estimates the visible dyed-phase color using channel-wise optical density: clear T100 contributes no dye and does not wash out the colored phase, while orange T200 and blue T400 combine as transparent colorants. This matches the reference footage better than treating the visibly separate clear dose as a white pigment or homogeneous diluent. Per-recipe phase corrections were measured from video 5 at 50% and 75% fill using the fixed crop `x=294..355, y=366..407` and HSV saturation above 35; they are empirical camera/material corrections, not physical color measurements. The continuous-video renderers update the liquid material by cumulative dispense order on each sampled animation frame; rejected vial 2 starts with the last colored tank used for vial 1. When the liquid mesh is vertically scaled for appearance, the renderer also stretches the stream toward the current liquid surface while keeping its nozzle end fixed. These geometry edits stay in the render session layer and do not modify the USD assets.

The USD scene supplies generic animated fill and stream geometry, not recipe-specific timing or a physically calibrated liquid simulation. Recipe-conditioned outputs are still approximations; the T200 orange and T400 blue source palette and opacity were matched to supplied stills, but mixed recipes have not yet met a realism-qualified match against the reference footage. Batch-render manifests mark single-frame samples with `label_quality=recipe_color_approximation` and list recipe components that are not rendered.

Continuous FillCam sequences only update the vial liquid color while the
animated dispensing stream is visible. During nozzle pauses, the last liquid
color is held instead of blending forward with the animation clock.

Single-frame color calibration renders are in `dataset/color_calibration/`.
This includes mixed-recipe pilots for recipes 5–7. They are hue sanity checks,
not realism-qualified matches: rendered liquid brightness, transparency, fill
level, and reflections still differ from the source footage. Do not treat this
model as production-calibrated. Previously rendered videos remain unchanged
until explicitly regenerated.
The source-palette sRGB values remain T200 `(0.45, 0.20, 0.00)` and T400
`(0.00, 0.29, 0.45)`. Recipe-conditioned FillCam renders use liquid opacity
`0.75` and a `0.25` vertical scale on the generic fill mesh to approximate the
transparency and short colored-liquid region visible in the reference vials.
The version-6 recipe-phase corrections for recipes 5 and 6 were remeasured at
this opacity using matched 50% and 75% fill samples. Corrections for recipes
7–14 are retained from the earlier opacity setting and have not yet been
revalidated at `0.75`. Changing these render settings requires recalibration.
The camera, lighting, and scene differ from the source footage, so the results
remain empirical visual approximations rather than calibrated
spectral/material measurements. Existing videos are unchanged until explicitly
regenerated.
Recipe 7 additionally receives a 10% linear-RGB neutral lift after its phase
correction. This represents the clear T100 portion of its pale-blue mixture,
which the colored-phase model otherwise omits.

For causal TCN training, build a timestep manifest with
`tools/build_causal_tcn_manifest.py`. It emits fixed-length, past-only frame
windows, explicit current and future-rejection targets, contamination onset,
and deterministic transition-level train/validation/test splits. The split is
assigned per clip before frames are emitted, preventing adjacent frames from
the same transition from crossing dataset boundaries:

```bash
python tools/build_causal_tcn_manifest.py \
  --dataset-root dataset/recipe_07_to_08_pass_and_fail_04ml_v1_10fps \
  --output dataset/recipe_07_to_08_causal_tcn.jsonl \
  --window-frames 32 \
  --horizon-frames 10
```

```bash
/isaac-sim/python.sh tools/render_batch1.py \
  --labels-csv data/video5_vial_recipes.csv \
  --out dataset/video5_recipe_color \
  --samples-per-vial 1 --subframes 8 --seed 7
```

Train/validation/test assignment is split by vial position within each recipe group so every recipe is represented in all three splits.

For a continuous two-vial sequence, use `tools/render_recipe_video.py` with a
JSON file containing exactly vial 1 and vial 2. The frame-label CSV follows
the source schema, while the recipe CSV records the generated timing. Accepted
mode renders both vials according to their intended recipes:

```bash
/isaac-sim/python.sh tools/render_recipe_video.py \
  --recipes data/two_vial_recipes.json \
  --out dataset/two_vial_accepted \
  --mode accepted --frames-per-vial 90 --gap-frames 6 --fps 30
```

Rejected mode carries the last colored tank used for vial 1 into the beginning
of vial 2's fill; clear T100 is not treated as a colored contaminant. The
labels remain the intended recipe labels and record the contamination amount
and tank:

```bash
/isaac-sim/python.sh tools/render_recipe_video.py \
  --recipes data/two_vial_recipes.json \
  --out dataset/two_vial_rejected \
  --mode rejected --residual-fraction 0.10 \
  --frames-per-vial 90 --gap-frames 6 --fps 30
```

To plan or render the ordered recipe-permutation dataset, use
`generate_recipe_permutations.py`. Ordered pairs are intentional: vial 1's
last recipe determines vial 2's contamination. The default matrix contains
90 accepted videos and 810 rejected videos (2–10 ml carryover). The default
30 fps renders every animation frame. Use a divisor of 30 such as `--fps 15`
to sample every second animation frame; this approximately halves render work
while preserving clip duration, with label timestamps and frame indices
adjusted to the sampled output. Use
`--dry-run` to write and inspect manifests without rendering; without it the
script renders PNGs, encodes MP4s, and verifies encoded frame counts:

```bash
/isaac-sim/python.sh tools/generate_recipe_permutations.py \
  --labels-csv data/video5_vial_recipes.csv \
  --out dataset/recipe_permutations --scenario both
```

For a smaller verified batch, repeat `--recipe-pair BEFORE:AFTER` as needed
and specify contamination levels. This renders one accepted and one rejected
video for the selected pair at 15 fps:

```bash
/isaac-sim/python.sh tools/generate_recipe_permutations.py \
  --labels-csv data/video5_vial_recipes.csv \
  --out dataset/recipe_pair_05_06_15fps \
  --scenario both --recipe-pair 5:6 \
  --contamination-ml 2 --fps 15
```

Existing MP4s cannot be reliably recolored in place: fluid color is mixed with
glass, highlights, reflections, and the animated fill geometry. Because the
render plans and animation timing are retained, regenerate into a new output
directory with the updated color model and validate the new videos before
promoting them; the existing datasets are not overwritten automatically.

## Quick start
Run with Isaac Sim's Python (`python.sh` in the Isaac Sim install folder):
```bash
# full line, fill camera, finished vial
<isaac-sim>/python.sh tools/render_example.py --camera FillCam --frame 686 --out fill.png
# full line, cap-seal camera (lens distortion applied automatically)
<isaac-sim>/python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
# a single station file
<isaac-sim>/python.sh tools/render_example.py --camera FillCam --frame 301 \
    --stage stations/fill_station/fill_station.usda --out fill_station.png
```
`apply_lens.py` can also be run on its own, with any Python that has `numpy` and `opencv-python`:
```bash
python tools/apply_lens.py cap_seal_render_768x576.png cameras/CapSealCam.json cap_seal_final.png
```

## Notes
- **Materials** are standard USD materials (UsdPreviewSurface). Glass and liquid are approximations, so you may want to adjust them for Isaac Sim's RTX renderer.
- **Labels on fill station parts:** they carry `userProperties:synthetic_class`, `synthetic_class_id` and `synthetic_instance_id` attributes, which can be used as labels.
- **Tested with:** Isaac Sim 6.0.
